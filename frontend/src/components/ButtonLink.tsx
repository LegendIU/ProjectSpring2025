import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';

interface ButtonLinkProps {
  to: string;
  children: ReactNode;
  variant?: 'primary' | 'ghost';
}

const ButtonLink = ({ to, children, variant = 'primary' }: ButtonLinkProps) => {
  return (
    <Link className={`button button--${variant}`} to={to}>
      {children}
    </Link>
  );
};

export default ButtonLink;
