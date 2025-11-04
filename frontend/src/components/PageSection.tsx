import type { ReactNode } from 'react';

interface PageSectionProps {
  id?: string;
  title: string;
  subtitle?: string;
  description?: string;
  align?: 'left' | 'center';
  children?: ReactNode;
}

const PageSection = ({
  id,
  title,
  subtitle,
  description,
  align = 'left',
  children
}: PageSectionProps) => {
  return (
    <section id={id} className={`section section--${align}`}>
      <header className="section__header">
        {subtitle ? <p className="section__subtitle">{subtitle}</p> : null}
        <h2 className="section__title">{title}</h2>
        {description ? <p className="section__description">{description}</p> : null}
      </header>
      {children ? <div className="section__body">{children}</div> : null}
    </section>
  );
};

export default PageSection;
